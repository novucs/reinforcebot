import React, {Component} from "react";
import {Container, Grid, Header, List, Segment} from "semantic-ui-react";


export default class Footer extends Component {
  render() {
    return (
      <Segment inverted vertical style={{padding: '5em 0em'}}>
        <Container>
          <Grid>
            <Grid.Row>
              <Grid.Column width={4}/>
              <Grid.Column width={8}>
                <Header as='h4' inverted>ReinforceBot</Header>
                <p>Automate anything with assisted creation of RL agents.</p>
                <List link inverted horizontal divided>
                  <List.Item as='a' href='start'>Quickstart</List.Item>
                  <List.Item as='a' href='mailto:william2.randall@live.uwe.ac.uk'>Contact Us</List.Item>
                </List>
              </Grid.Column>
            </Grid.Row>
          </Grid>
        </Container>
      </Segment>
    )
  }
}